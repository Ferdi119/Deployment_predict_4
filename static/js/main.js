$(document).ready(function($) {

    'use strict';

/*=======================================
    PRELOADER                     
======================================= */
    $(window).on('load', function() {
        $('#preloader').fadeOut('slow', function() {
            $(this).remove();
        });
    });
	
	
/* =======================================
    single Page Nav
=======================================*/
    // The actual plugin
    $('.single-page-nav').singlePageNav({
        offset: $('.single-page-nav').outerHeight(),
        filter: ':not(.external)',
        updateHash: true,
    });


/* =======================================
    collapse the navbar on scroll
=======================================*/
    // jQuery to collapse the navbar on scroll
    $(window).on('scroll', function() {
        if ($(".navbar").offset().top > 50) {
            $(".navbar-fixed-top").addClass("top-nav-collapse");
        } else {
            $(".navbar-fixed-top").removeClass("top-nav-collapse");
        }
    });


/* =======================================
    WOW ANIMATION
======================================= */
    var wow = new WOW({
        mobile: false
    });
    wow.init();

/*=======================================
    Scroll Top
=======================================*/
    $("#scrollup").on('click', function() {
        $('html,body').animate({
            'scrollTop': '0'
        }, 4000);
        return false;
    });


/*=======================================
    TESTIMONIAL CAROUSE                 
======================================= */
    $('#testimonial').owlCarousel({
        loop: true,
        center: true,
        stagePadding: 10,
        responsiveClass: true,
        responsive: {
            0: {
                items: 1,
                nav: false,
                dots: true
            },
            767: {
                items: 3,
                dots: false,
                nav: true,
                navText: [" <i class='arrow_carrot-left'></i> ", " <i class='arrow_carrot-right'></i> "]
            }
        }
    });


/* =======================================
    Gallery Section : mixItUp jQuery 
=======================================*/
    $('.project-list').mixItUp({
        animation: {
            effects: 'fade stagger(50ms)',
            reverseOut: true,
            staggerSequence: function(i) {
                return (2 * i) - (5 * ((i / 3) - ((1 / 3) * (i % 3))));
            }
        }
    });


});